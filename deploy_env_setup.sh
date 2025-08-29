#!/bin/bash
# Railway Environment Setup Script
echo "🔧 Setting up Railway production environment variables..."

# Core application settings
railway variables set ENVIRONMENT=production
railway variables set DEBUG=false
railway variables set APP_VERSION=2.0.0
railway variables set PORT=8000
railway variables set HOST=0.0.0.0
railway variables set WORKERS=1

# Railway optimization settings
railway variables set RAILWAY_OPTIMIZED=true
railway variables set RAILWAY_PRODUCTION=true

# Database (Railway will provide DATABASE_URL automatically)
echo "✅ DATABASE_URL will be provided automatically by Railway PostgreSQL service"

# AI API Keys (you need to set these with your actual keys)
echo "⚠️  You need to set these manually with your actual API keys:"
echo "  railway variables set OPENAI_API_KEY=your-actual-openai-api-key"
echo "  railway variables set LANGCHAIN_API_KEY=your-actual-langchain-api-key (optional)"

# Security settings
railway variables set SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
railway variables set JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# CORS settings (update with your actual domain)
railway variables set ALLOWED_ORIGINS="https://your-railway-domain.up.railway.app,http://localhost:5173"

# Performance settings
railway variables set CACHE_ENABLED=true
railway variables set COMPRESSION_ENABLED=true
railway variables set ENABLE_RATE_LIMITING=true
railway variables set RATE_LIMIT_PER_MINUTE=60

# Logging
railway variables set LOG_LEVEL=INFO
railway variables set ENABLE_ACCESS_LOG=true

# Health checks
railway variables set HEALTH_CHECK_INTERVAL=30

echo "✅ Basic environment variables set!"
echo "🔑 Don't forget to set your API keys manually!"