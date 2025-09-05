#!/bin/bash

# 🚀 CrediLinq AI Platform - Quick Deploy Script
# Run this script in your terminal to deploy to Railway

echo "🚀 CrediLinq AI Content Platform with LangGraph - Deployment"
echo "============================================================"

# Check if logged into Railway
echo "🔍 Checking Railway login status..."
if ! railway whoami &>/dev/null; then
    echo "❌ Not logged into Railway"
    echo "📋 Please run: railway login"
    echo "   This will open a browser for authentication"
    exit 1
fi

echo "✅ Railway login verified"
echo ""

# Show current project status
echo "📊 Current project status:"
railway whoami
echo ""

# Deploy the main service
echo "🚀 Deploying CrediLinq Backend + LangGraph Service..."
echo "   This includes:"
echo "   • FastAPI Backend (Port 8000)"
echo "   • LangGraph Workflows (Port 8001)" 
echo "   • AI Agent Orchestration"
echo "   • Campaign Management"
echo ""

read -p "Ready to deploy? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Starting deployment..."
    railway up --detach
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Deployment initiated successfully!"
        echo ""
        echo "📊 Check deployment status:"
        echo "   railway logs --follow"
        echo ""
        echo "📈 Monitor your services:"
        echo "   • Railway Dashboard: https://railway.app/dashboard"
        echo "   • API Health: https://your-app.railway.app/health/railway"
        echo "   • LangGraph Studio: https://smith.langchain.com/studio/"
        echo ""
        echo "🔗 Once deployed, connect LangGraph Studio to:"
        echo "   https://your-app.railway.app:8001"
        echo ""
    else
        echo "❌ Deployment failed. Check the output above for errors."
        exit 1
    fi
else
    echo "Deployment cancelled."
fi