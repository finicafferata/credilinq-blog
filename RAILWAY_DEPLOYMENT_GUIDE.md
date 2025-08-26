# Railway Deployment Guide for CrediLinq AI Platform

## Quick Fix for Current Crash

Based on your crash pattern (successful initialization followed by crash), here are immediate steps:

### 1. Deploy with Railway Mode

Set these environment variables in Railway dashboard:

```bash
RAILWAY_MODE=true
MINIMAL_MODE=true
ENVIRONMENT=production
```

### 2. Use Optimized Requirements

Update your Railway service to use the lighter requirements file:
- Change build command to use `requirements-railway.txt` instead of `requirements.txt`

### 3. Update Dockerfile

Replace your current Dockerfile with `Dockerfile.railway` for reduced memory usage:

```bash
# In Railway dashboard, set:
# Dockerfile Path: Dockerfile.railway
```

## Environment Variables Setup

### Required Variables
```bash
DATABASE_URL=postgresql://[railway-provided-url]
OPENAI_API_KEY=your_openai_api_key
PORT=[auto-set-by-railway]
```

### Optional but Recommended
```bash
RAILWAY_MODE=true
MINIMAL_MODE=true
ENVIRONMENT=production
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

## Debugging Your Current Crash

### Step 1: Check Logs
```bash
railway logs
```
Look for:
- Memory limit exceeded
- Database connection errors
- Port binding issues
- Import/dependency errors

### Step 2: Run Diagnostic Script
```bash
# Locally first:
python scripts/railway_debug.py

# Then deploy with debug mode:
# Set RAILWAY_MODE=true and redeploy
```

### Step 3: Test Health Endpoints
After deployment, test:
- `https://your-app.railway.app/health/live` - Should return 200
- `https://your-app.railway.app/ping` - Simple ping test
- `https://your-app.railway.app/test` - Environment variable test

## Common Crash Causes & Solutions

### 1. Memory Limit Exceeded
**Symptoms**: App crashes after initialization, logs show high memory usage

**Solution**: 
- Use `requirements-railway.txt` (lighter dependencies)
- Set `RAILWAY_MODE=true`
- Increase memory limit in Railway dashboard

### 2. Database Connection Issues
**Symptoms**: Database errors in logs, health checks fail

**Solution**:
- Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/db`
- Railway uses SSL - ensure SSL is enabled
- Check database service is running in Railway

### 3. Port Binding Issues
**Symptoms**: "Address already in use", "Port binding failed"

**Solution**:
- Never hardcode port - always use `os.getenv("PORT")`
- Bind to `0.0.0.0:${PORT}`, not `localhost`

### 4. AI/ML Dependencies Crash
**Symptoms**: ImportError, memory spikes during import

**Solution**:
- Use `MINIMAL_MODE=true` to load only essential features
- Import heavy dependencies lazily
- Consider using Railway's persistent disk for model caching

## Railway-Specific Optimizations

### 1. Health Check Configuration
```toml
# railway.toml
[deploy]
healthcheckPath = "/health/live"
healthcheckTimeout = 60
```

### 2. Resource Limits
```toml
# railway.toml
memoryLimit = "1GB"
cpuLimit = "1000m"
```

### 3. Startup Command
```toml
# railway.toml
startCommand = "python /app/scripts/start.py"
```

## Monitoring and Maintenance

### Health Check Endpoints
- `/health/live` - Liveness probe (minimal)
- `/health/ready` - Readiness probe (with DB check)
- `/ping` - Simple ping test

### Logs Analysis
```bash
# View recent logs
railway logs

# Follow logs in real-time
railway logs --follow

# Filter for errors
railway logs | grep -i error
```

### Performance Monitoring
- Monitor memory usage in Railway dashboard
- Set up alerts for crashes/restarts
- Use `/health/metrics` endpoint for detailed metrics

## Troubleshooting Steps

### If App Still Crashes:

1. **Deploy Minimal Version**:
   ```bash
   MINIMAL_MODE=true
   RAILWAY_MODE=true
   ```

2. **Check Resource Usage**:
   - Memory consumption
   - CPU usage during startup
   - Disk usage for dependencies

3. **Isolate the Issue**:
   - Test with only health endpoints
   - Gradually enable features
   - Monitor logs for specific failure points

4. **Database Issues**:
   ```bash
   # Test connection manually
   railway run python -c "
   import os
   import psycopg2
   conn = psycopg2.connect(os.getenv('DATABASE_URL'))
   print('DB connection successful')
   "
   ```

### Common Error Patterns:

#### "Process exited with code 137"
- **Cause**: Memory limit exceeded
- **Fix**: Reduce memory usage, increase Railway memory limit

#### "Process exited with code 1" 
- **Cause**: Application error during startup
- **Fix**: Check logs, fix import errors, verify environment variables

#### Health check failures
- **Cause**: App not responding on correct port
- **Fix**: Verify port binding, health endpoint accessibility

## Next Steps for Debugging

1. Deploy with `RAILWAY_MODE=true` first
2. Monitor logs during startup
3. Test health endpoints
4. Gradually enable full features if minimal version works
5. Use diagnostic script for detailed environment analysis

This approach will help identify if the issue is:
- Memory/resource limits
- Database connectivity
- Dependency loading
- Port/networking configuration