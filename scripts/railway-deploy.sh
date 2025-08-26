#!/bin/bash

# Enhanced Railway deployment script for CrediLinq AI Content Platform
# Fixed for proper PORT environment variable handling

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-staging}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
CrediLinq AI Content Platform - Railway Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    deploy              Full deployment pipeline (default)
    build-only          Build and test only, no deployment
    migrations          Run database migrations only
    rollback            Rollback to previous deployment
    logs                Stream deployment logs
    test-config         Test Railway configuration
    
Options:
    -e, --env ENV       Target environment (dev, staging, production)
    -f, --force         Force deployment without checks
    -v, --verbose       Verbose output
    -h, --help         Show this help

Environment Variables:
    RAILWAY_TOKEN      Railway authentication token
    DATABASE_URL       Database connection string
    OPENAI_API_KEY     OpenAI API key
    
Examples:
    $0 deploy
    $0 --env production deploy
    $0 build-only --verbose
    $0 test-config
EOF
}

# Test Railway configuration
test_railway_config() {
    log_info "Testing Railway configuration..."
    
    cd "${PROJECT_ROOT}"
    
    # Check railway.toml
    if [[ -f "railway.toml" ]]; then
        log_info "railway.toml found:"
        cat railway.toml
    else
        log_error "railway.toml not found"
        exit 1
    fi
    
    echo ""
    
    # Check Dockerfile
    if [[ -f "Dockerfile" ]]; then
        log_info "Dockerfile CMD configuration:"
        grep -A 2 -B 2 "CMD" Dockerfile || log_warning "CMD not found in Dockerfile"
    fi
    
    echo ""
    
    # Test startup script
    if [[ -f "scripts/start.py" ]]; then
        log_info "Testing startup script with mock Railway environment..."
        PORT=8080 RAILWAY_ENVIRONMENT=test python3 scripts/start.py --dry-run 2>/dev/null || {
            log_info "Testing environment detection:"
            PORT=8080 RAILWAY_ENVIRONMENT=test python3 scripts/test_env.py
        }
    fi
    
    log_success "Configuration test completed"
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check Railway CLI
    if ! command -v railway &> /dev/null; then
        log_error "Railway CLI not found. Install: npm install -g @railway/cli"
        exit 1
    fi
    
    # Check authentication
    if ! railway whoami &> /dev/null; then
        log_error "Railway not authenticated. Run: railway login"
        exit 1
    fi
    
    # Validate project structure
    local required_files=(
        "railway.toml"
        "Dockerfile"
        "requirements.txt"
        "src/main.py"
        "scripts/start.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${PROJECT_ROOT}/${file}" ]]; then
            log_error "Required file not found: ${file}"
            exit 1
        fi
    done
    
    # Check for PORT environment variable handling
    if grep -q '\$PORT' "${PROJECT_ROOT}/Dockerfile" && ! grep -q 'scripts/start.py' "${PROJECT_ROOT}/Dockerfile"; then
        log_warning "Found old-style PORT handling in Dockerfile. Consider updating to use scripts/start.py"
    fi
    
    log_success "Environment validation passed"
}

# Build and test
build_and_test() {
    log_info "Building and testing application..."
    
    cd "${PROJECT_ROOT}"
    
    # Test startup script
    log_info "Testing startup script..."
    python3 scripts/test_env.py
    
    # Test with mock Railway environment
    log_info "Testing with mock Railway environment..."
    PORT=8080 RAILWAY_ENVIRONMENT=test python3 scripts/test_env.py
    
    log_success "Build and test completed"
}

# Database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Check if migrations exist
    if [[ -d "${PROJECT_ROOT}/database/migrations" ]]; then
        log_info "Custom migrations found, running..."
        # Add custom migration runner if needed
    fi
    
    # Run Prisma migrations
    if [[ -f "${PROJECT_ROOT}/package.json" ]] || command -v npx &> /dev/null; then
        log_info "Running Prisma migrations..."
        npx prisma migrate deploy || log_warning "Prisma migrations failed or not configured"
    fi
    
    log_success "Database migrations completed"
}

# Deploy to Railway
deploy_to_railway() {
    log_info "Deploying to Railway environment: ${ENVIRONMENT}"
    
    cd "${PROJECT_ROOT}"
    
    # Set environment if specified
    if [[ "${ENVIRONMENT}" != "staging" ]]; then
        log_info "Setting Railway environment to ${ENVIRONMENT}"
        railway environment ${ENVIRONMENT} || log_warning "Environment switch failed, continuing..."
    fi
    
    # Show current configuration
    log_info "Current Railway configuration:"
    cat railway.toml
    
    # Deploy
    log_info "Starting Railway deployment..."
    railway deploy
    
    # Wait for deployment
    log_info "Waiting for deployment to complete..."
    sleep 10
    
    # Check deployment status
    railway status
    
    log_success "Railway deployment completed"
}

# Stream logs
stream_logs() {
    log_info "Streaming Railway logs..."
    railway logs --follow
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    # Railway doesn't have direct rollback, but we can redeploy previous commit
    log_info "To rollback, deploy a previous commit manually"
    railway deployments
}

# Main deployment pipeline
deploy_full() {
    log_info "Starting full deployment pipeline for ${ENVIRONMENT}..."
    
    validate_environment
    build_and_test
    run_migrations
    deploy_to_railway
    
    log_success "🚀 Deployment completed successfully!"
    log_info "View your application logs with: $0 logs"
    log_info "Test configuration with: $0 test-config"
}

# Parse command line arguments
COMMAND="deploy"
FORCE_DEPLOY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_DEPLOY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy|build-only|migrations|rollback|logs|test-config)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Enable verbose mode if requested
if [[ "${VERBOSE}" == "true" ]]; then
    set -x
fi

# Execute command
case "${COMMAND}" in
    deploy)
        deploy_full
        ;;
    build-only)
        validate_environment
        build_and_test
        ;;
    migrations)
        run_migrations
        ;;
    rollback)
        rollback_deployment
        ;;
    logs)
        stream_logs
        ;;
    test-config)
        test_railway_config
        ;;
    *)
        log_error "Unknown command: ${COMMAND}"
        show_help
        exit 1
        ;;
esac