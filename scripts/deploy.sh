#!/bin/bash

# Production Deployment Script for CrediLinq AI Content Platform
# This script handles the complete deployment process

set -e  # Exit on any error

# Configuration
APP_NAME="credilinq-content-platform"
DOCKER_IMAGE="credilinq/content-platform"
CONTAINER_NAME="credilinq-app"
BACKUP_DIR="/var/backups/credilinq"
LOG_FILE="/var/log/credilinq-deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}INFO:${NC} $1"
}

log_success() {
    log "${GREEN}SUCCESS:${NC} $1"
}

log_warning() {
    log "${YELLOW}WARNING:${NC} $1"
}

log_error() {
    log "${RED}ERROR:${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available disk space (minimum 2GB)
    available_space=$(df / | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 2097152 ]]; then
        log_error "Insufficient disk space. At least 2GB required"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Backup current deployment
backup_current() {
    log_info "Creating backup of current deployment..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR/$(date +%Y%m%d_%H%M%S)"
    BACKUP_PATH="$BACKUP_DIR/$(date +%Y%m%d_%H%M%S)"
    
    # Backup database
    if docker ps | grep -q credilinq-postgres; then
        log_info "Backing up database..."
        docker exec credilinq-postgres pg_dump -U credilinq credilinq > "$BACKUP_PATH/database.sql"
        log_success "Database backup created"
    fi
    
    # Backup application data
    if docker volume ls | grep -q credilinq; then
        log_info "Backing up application data..."
        docker run --rm -v credilinq_app-logs:/source -v "$BACKUP_PATH":/backup alpine tar czf /backup/app-logs.tar.gz -C /source .
        docker run --rm -v credilinq_app-cache:/source -v "$BACKUP_PATH":/backup alpine tar czf /backup/app-cache.tar.gz -C /source .
        log_success "Application data backup created"
    fi
    
    echo "$BACKUP_PATH" > /tmp/credilinq-backup-path
    log_success "Backup completed: $BACKUP_PATH"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Get version from git or default
    VERSION=${1:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}
    BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    docker build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        -t "$DOCKER_IMAGE:$VERSION" \
        -t "$DOCKER_IMAGE:latest" \
        .
    
    log_success "Docker image built: $DOCKER_IMAGE:$VERSION"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    timeout=60
    while ! docker exec credilinq-postgres pg_isready -U credilinq -d credilinq &>/dev/null; do
        if [[ $timeout -le 0 ]]; then
            log_error "Database is not ready after 60 seconds"
            exit 1
        fi
        log_info "Waiting for database..."
        sleep 5
        ((timeout-=5))
    done
    
    # Run Prisma migrations
    docker exec "$CONTAINER_NAME" npx prisma migrate deploy || {
        log_warning "Prisma migrations failed, continuing with manual migration..."
        
        # Run manual migrations if available
        if [ -d "./database/migrations" ]; then
            for migration in ./database/migrations/*.sql; do
                if [ -f "$migration" ]; then
                    log_info "Running migration: $(basename "$migration")"
                    docker exec credilinq-postgres psql -U credilinq -d credilinq -f "/docker-entrypoint-initdb.d/$(basename "$migration")"
                fi
            done
        fi
    }
    
    log_success "Database migrations completed"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health/ready &>/dev/null; then
            log_success "Application is healthy"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Application failed health check after $max_attempts attempts"
    return 1
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    # Get backup path
    if [ -f /tmp/credilinq-backup-path ]; then
        BACKUP_PATH=$(cat /tmp/credilinq-backup-path)
        
        # Stop current containers
        docker-compose down
        
        # Restore database
        if [ -f "$BACKUP_PATH/database.sql" ]; then
            log_info "Restoring database..."
            docker-compose up -d postgres
            sleep 10
            docker exec credilinq-postgres psql -U credilinq -d credilinq < "$BACKUP_PATH/database.sql"
        fi
        
        # Restore application data
        if [ -f "$BACKUP_PATH/app-logs.tar.gz" ]; then
            docker run --rm -v credilinq_app-logs:/target -v "$BACKUP_PATH":/backup alpine tar xzf /backup/app-logs.tar.gz -C /target
        fi
        
        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
    fi
}

# Cleanup old backups
cleanup_backups() {
    log_info "Cleaning up old backups..."
    
    # Keep last 7 days of backups
    find "$BACKUP_DIR" -type d -name "20*" -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    log_success "Backup cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of $APP_NAME..."
    
    # Trap for cleanup on failure
    trap rollback ERR
    
    # Pre-deployment checks
    check_root
    check_requirements
    
    # Create backup
    backup_current
    
    # Build new image
    build_image "$1"
    
    # Deploy with zero-downtime strategy
    log_info "Deploying application..."
    
    # Start new services
    docker-compose pull
    docker-compose up -d --remove-orphans
    
    # Wait for services to be ready
    sleep 30
    
    # Run migrations
    run_migrations
    
    # Health check
    if health_check; then
        log_success "Deployment completed successfully!"
        
        # Cleanup
        docker system prune -f
        cleanup_backups
        
        # Send notification (if configured)
        if command -v mail &> /dev/null && [ -n "$NOTIFICATION_EMAIL" ]; then
            echo "CrediLinq deployment completed successfully at $(date)" | mail -s "Deployment Success" "$NOTIFICATION_EMAIL"
        fi
        
    else
        log_error "Health check failed, rolling back..."
        rollback
        exit 1
    fi
}

# Script options
case "${1:-deploy}" in
    "deploy")
        deploy "$2"
        ;;
    "rollback")
        rollback
        ;;
    "health")
        health_check
        ;;
    "backup")
        backup_current
        ;;
    "build")
        build_image "$2"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|backup|build} [version]"
        exit 1
        ;;
esac