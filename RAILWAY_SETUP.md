# Railway Deployment Setup Guide

## Critical Environment Variables

Set these environment variables in your Railway dashboard:

### Required Variables
```bash
DATABASE_URL=postgresql://username:password@host:port/database
DATABASE_URL_DIRECT=postgresql://username:password@host:port/database
ENVIRONMENT=production
```

### Security Variables
```bash
SECRET_KEY=your-super-secure-secret-key-here
JWT_SECRET=your-jwt-secret-key-here
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=your-secure-admin-password
```

### AI Provider (Choose One)
```bash
# Option 1: OpenAI
PRIMARY_AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here

# Option 2: Google/Gemini
PRIMARY_AI_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
```

### Railway Optimization Variables
```bash
PORT=8080
HOST=0.0.0.0
WORKERS=1
DEBUG=false
ENABLE_ANALYTICS=false
ENABLE_CACHE=false
RAILWAY_OPTIMIZED=true
```

## Railway Service Configuration

1. **Memory Limit**: Set to at least 1GB (2GB recommended for production)
2. **CPU Limit**: 1000m or higher
3. **Health Check Path**: `/health/live`
4. **Health Check Timeout**: 120 seconds

## Deployment Steps

1. Set all required environment variables in Railway dashboard
2. Connect your repository to Railway
3. Railway will automatically deploy using the Railway-optimized configuration
4. Monitor deployment logs for any issues
5. Test health endpoint: `https://your-app.railway.app/health/live`

## Troubleshooting

### Common Issues

1. **Startup Timeout**
   - Increase health check timeout to 120s
   - Check memory limits (increase if needed)
   - Verify all required environment variables are set

2. **Database Connection Issues**
   - Verify DATABASE_URL format: `postgresql://user:pass@host:port/db`
   - Check if database service is running
   - Ensure database allows connections from Railway IPs

3. **Memory Issues**
   - Increase memory limit to 2GB for production
   - Use Railway-optimized version (automatically enabled)

4. **AI Service Issues**
   - Verify API keys are correctly set
   - Check PRIMARY_AI_PROVIDER matches available keys
   - AI features will be limited without valid API keys

## Monitoring

- Health Check: `GET /health/live`
- Simple Ping: `GET /ping`
- API Docs: `GET /docs`
- Status: `GET /` (shows optimization level)

## Production Considerations

1. **Environment Variables**: Never commit secrets to code
2. **Database**: Use Railway's PostgreSQL service or external provider
3. **Monitoring**: Set up Railway's monitoring and alerts
4. **Scaling**: Start with 1GB memory, scale up if needed
5. **Backups**: Ensure database backups are configured