#!/bin/bash

# CrediLinq Content Agent - Railway Deployment Script
# This script automates the deployment process to Railway

set -e  # Exit on any error

echo "🚀 CrediLinq Content Agent - Railway Deployment"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    print_error "Railway CLI is not installed. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

print_status "Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    print_error "You are not logged in to Railway. Please run 'railway login' first."
    exit 1
fi

print_success "Railway CLI authenticated successfully"

# Check if we're in a Railway project
if ! railway status &> /dev/null; then
    print_warning "Not currently linked to a Railway project."
    echo "Would you like to:"
    echo "1) Link to existing project"
    echo "2) Create new project"
    read -p "Enter choice (1 or 2): " choice
    
    if [ "$choice" = "1" ]; then
        railway link
    elif [ "$choice" = "2" ]; then
        railway init
    else
        print_error "Invalid choice. Exiting."
        exit 1
    fi
fi

print_status "Checking current Railway project status..."
railway status

# Add PostgreSQL if not exists
print_status "Setting up PostgreSQL database..."
if ! railway services | grep -q "postgresql"; then
    print_status "Adding PostgreSQL service..."
    railway add postgresql
    print_success "PostgreSQL service added"
else
    print_success "PostgreSQL service already exists"
fi

# Generate secure secrets if not provided
print_status "Checking environment variables..."

# Generate SECRET_KEY if not set
if ! railway variables get SECRET_KEY &> /dev/null; then
    SECRET_KEY=$(openssl rand -base64 32)
    railway variables set SECRET_KEY="$SECRET_KEY"
    print_success "Generated and set SECRET_KEY"
fi

# Generate JWT_SECRET if not set
if ! railway variables get JWT_SECRET &> /dev/null; then
    JWT_SECRET=$(openssl rand -base64 32)
    railway variables set JWT_SECRET="$JWT_SECRET"
    print_success "Generated and set JWT_SECRET"
fi

# Set other production variables
print_status "Setting production environment variables..."

railway variables set ENVIRONMENT="production"
railway variables set API_VERSION="2.0.0"
railway variables set WORKERS="1"
railway variables set RATE_LIMIT_PER_MINUTE="100"
railway variables set RATE_LIMIT_PER_HOUR="2000"
railway variables set ENABLE_MONITORING="true"
railway variables set LOG_LEVEL="INFO"
railway variables set RAILWAY_PRODUCTION="true"

# Check critical API keys
if ! railway variables get OPENAI_API_KEY &> /dev/null; then
    print_warning "OPENAI_API_KEY not set. AI features will not work."
    read -p "Enter your OpenAI API key (or press Enter to skip): " openai_key
    if [ ! -z "$openai_key" ]; then
        railway variables set OPENAI_API_KEY="$openai_key"
        print_success "OpenAI API key set"
    fi
fi

print_status "Deploying backend service..."
railway up --detach

print_success "Backend deployment initiated!"

# Wait for deployment to complete
print_status "Waiting for deployment to complete..."
sleep 10

# Check deployment status
print_status "Checking deployment status..."
railway logs --tail

print_success "Deployment process completed!"
print_status "Next steps:"
echo "1. Check logs: railway logs"
echo "2. Get service URL: railway domain"
echo "3. Run database migrations"
echo "4. Deploy frontend service"

echo ""
print_success "Backend service should be available at: $(railway domain 2>/dev/null || echo 'Check Railway dashboard for URL')"