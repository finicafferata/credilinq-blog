# Railway Deployment Guide

## Overview

This guide covers deploying the CrediLinq AI Content Platform to Railway with proper PORT environment variable handling and optimized configuration.

## Railway Configuration Files

### 1. railway.toml
```toml
[build]
command = "pip install -r requirements.txt"

[deploy]
startCommand = "python scripts/start.py"
```

### 2. Dockerfile
The Dockerfile is optimized for Railway with:
- Multi-stage build for smaller images
- Non-root user for security
- Railway-specific startup script
- Proper environment variable handling

### 3. Startup Script (scripts/start.py)
- Automatically detects Railway environment
- Properly handles PORT environment variable expansion
- Railway-specific optimizations (single worker, logging)
- Fallback defaults for local development

## Environment Variables

### Required Variables
```env
DATABASE_URL=postgresql://...
OPENAI_API_KEY=sk-...
```

### Optional Variables
```env
# Server configuration (Railway sets PORT automatically)
HOST=0.0.0.0
WORKERS=1

# Application configuration
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret

# Database
DB_CONNECTION_TIMEOUT=30
DB_MAX_RETRIES=3

# API Keys
LANGCHAIN_API_KEY=ls_...
```

## Deployment Steps

### 1. Initial Setup
1. Connect your GitHub repository to Railway
2. Set required environment variables in Railway dashboard
3. Deploy will start automatically

### 2. Environment Variable Configuration
Railway automatically provides:
- `PORT` - Dynamic port assignment
- `RAILWAY_ENVIRONMENT` - Environment detection
- `RAILWAY_PROJECT_ID`, `RAILWAY_SERVICE_ID` - Project identifiers

### 3. Database Setup
For PostgreSQL on Railway:
1. Add PostgreSQL service to your project
2. Railway will automatically set `DATABASE_URL`
3. Run database migrations after deployment

### 4. Verification
Check deployment logs for:
```
Starting CrediLinq AI Platform...
Port: 8000 (or Railway-assigned port)
Railway Environment: True
```

## Troubleshooting

### PORT Variable Issues
If you see: `Error: Invalid value for '--port': '$PORT' is not a valid integer`

**Solution**: The new configuration uses `scripts/start.py` which properly handles environment variables.

### Common Issues
1. **Build failures**: Check `requirements.txt` compatibility
2. **Database connection**: Verify `DATABASE_URL` is set
3. **Missing dependencies**: Ensure all Python packages are in requirements.txt

### Debug Commands
Use the test environment script:
```bash
# In Railway console or locally
python scripts/test_env.py
```

This will show:
- All environment variables
- PORT value and type
- Command construction
- Railway environment detection

## Best Practices

### Performance
- Single worker recommended for Railway (memory efficiency)
- Use Railway's built-in monitoring
- Enable health checks in Dockerfile

### Security
- Use environment variables for all secrets
- Non-root user in Docker container
- Security headers enabled in FastAPI middleware

### Monitoring
- Railway provides built-in logs and metrics
- Application health check at `/health/live`
- Structured logging for better debugging

## Local vs Railway Differences

| Aspect | Local Development | Railway Production |
|--------|-------------------|-------------------|
| Port | 8000 (default) | Dynamic (Railway-assigned) |
| Workers | 4 (configurable) | 1 (optimized) |
| Reload | Enabled | Disabled |
| Logging | DEBUG/INFO | INFO/WARNING |
| Database | Local PostgreSQL | Railway PostgreSQL |

## Migration from Previous Setup

If migrating from manual uvicorn commands:

1. **Old railway.toml**:
   ```toml
   startCommand = "uvicorn src.main:app --host 0.0.0.0 --port $PORT"
   ```

2. **New railway.toml**:
   ```toml
   startCommand = "python scripts/start.py"
   ```

3. **Benefits**:
   - Proper environment variable handling
   - Railway-specific optimizations
   - Better error handling and logging
   - Consistent behavior across environments

## Support

For deployment issues:
1. Check Railway logs in dashboard
2. Run `python scripts/test_env.py` for debugging
3. Verify all required environment variables are set
4. Check database connectivity and migrations