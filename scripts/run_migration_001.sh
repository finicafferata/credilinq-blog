#!/bin/bash

# Database Migration Runner for 001_add_execution_planning.sql
# This script safely runs the migration with proper error handling and verification

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Default database connection settings (can be overridden by environment variables)
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-credilinq}
DB_USER=${DB_USER:-postgres}

# Check if migration file exists
MIGRATION_FILE="database/migrations/001_add_execution_planning.sql"
if [ ! -f "$MIGRATION_FILE" ]; then
    print_error "Migration file not found: $MIGRATION_FILE"
    print_status "Please ensure you're running this from the project root directory"
    exit 1
fi

print_status "Starting Migration 001: Execution Planning Tables"
print_status "Database: $DB_NAME on $DB_HOST:$DB_PORT as user $DB_USER"

# Step 1: Test database connection
print_status "Testing database connection..."
if psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -c "SELECT version();" > /dev/null 2>&1; then
    print_success "Database connection successful"
else
    print_error "Cannot connect to database"
    print_status "Please check your database connection settings:"
    print_status "  DB_HOST=$DB_HOST"
    print_status "  DB_PORT=$DB_PORT" 
    print_status "  DB_NAME=$DB_NAME"
    print_status "  DB_USER=$DB_USER"
    exit 1
fi

# Step 2: Create backup of existing schema (optional but recommended)
print_status "Creating schema backup..."
BACKUP_FILE="database/backups/pre_migration_001_$(date +%Y%m%d_%H%M%S).sql"
mkdir -p "database/backups"

if pg_dump -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" --schema-only > "$BACKUP_FILE" 2>/dev/null; then
    print_success "Schema backup created: $BACKUP_FILE"
else
    print_warning "Could not create schema backup (continuing anyway)"
fi

# Step 3: Check if tables already exist
print_status "Checking for existing tables..."
EXISTING_TABLES=$(psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -t -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('execution_plans', 'agent_dependencies', 'workflow_state_live');
" | tr -d ' ' | grep -v '^$' || true)

if [ ! -z "$EXISTING_TABLES" ]; then
    print_warning "Some tables already exist:"
    echo "$EXISTING_TABLES"
    read -p "Continue with migration? This may modify existing tables. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Migration cancelled"
        exit 0
    fi
fi

# Step 4: Run the migration
print_status "Running migration..."
if psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -f "$MIGRATION_FILE"; then
    print_success "Migration completed successfully"
else
    print_error "Migration failed"
    exit 1
fi

# Step 5: Verify migration results
print_status "Verifying migration results..."

# Check if all tables were created
CREATED_TABLES=$(psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -t -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('execution_plans', 'agent_dependencies', 'workflow_state_live')
ORDER BY table_name;
" | tr -d ' ' | grep -v '^$')

EXPECTED_TABLES="agent_dependencies
execution_plans
workflow_state_live"

if [ "$CREATED_TABLES" = "$EXPECTED_TABLES" ]; then
    print_success "All tables created successfully"
else
    print_error "Some tables may not have been created correctly"
    print_status "Expected: execution_plans, agent_dependencies, workflow_state_live"
    print_status "Found: $CREATED_TABLES"
fi

# Check indexes
print_status "Verifying indexes..."
INDEX_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -t -c "
SELECT COUNT(*) 
FROM pg_indexes 
WHERE tablename IN ('execution_plans', 'agent_dependencies', 'workflow_state_live');
" | tr -d ' ')

if [ "$INDEX_COUNT" -gt 0 ]; then
    print_success "Indexes created: $INDEX_COUNT indexes"
else
    print_warning "No indexes found (this may indicate an issue)"
fi

# Check functions and triggers
print_status "Verifying functions and triggers..."
FUNCTION_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -t -c "
SELECT COUNT(*) 
FROM pg_proc 
WHERE proname = 'update_updated_at_column';
" | tr -d ' ')

if [ "$FUNCTION_EXISTS" -gt 0 ]; then
    print_success "Update function created successfully"
else
    print_warning "Update function may not have been created"
fi

# Step 6: Show table information
print_status "Migration Summary:"
psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -c "
SELECT 
    t.table_name,
    t.table_type,
    COALESCE(s.n_tup_ins, 0) as row_count,
    pg_size_pretty(pg_total_relation_size(t.table_name::regclass)) as size
FROM information_schema.tables t
LEFT JOIN pg_stat_user_tables s ON t.table_name = s.relname
WHERE t.table_schema = 'public' 
  AND t.table_name IN ('execution_plans', 'agent_dependencies', 'workflow_state_live')
ORDER BY t.table_name;
"

print_success "Migration 001 completed successfully!"
print_status "Next steps:"
print_status "1. Test the new tables with some sample data"
print_status "2. Update your application code to use the new tables"
print_status "3. Run the Master Planner Agent implementation"

# Step 7: Optional - Show table structure
read -p "Would you like to see the table structure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Table structures:"
    for table in execution_plans agent_dependencies workflow_state_live; do
        echo -e "\n${BLUE}=== $table ===${NC}"
        psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER" -c "\d $table"
    done
fi

print_success "All done! ✨"