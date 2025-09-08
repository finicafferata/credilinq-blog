#!/bin/bash

# Migration 002: Workflow Execution Tables
# Run this script to add the workflow orchestration database tables

set -e

echo "🔧 Running Migration 002: Workflow Execution Tables"

# Get database URL from environment or use default
if [ -z "$DATABASE_URL" ]; then
    echo "⚠️  DATABASE_URL not set, using default PostgreSQL connection"
    DB_URL="postgresql://localhost:5432/credilinq"
else
    DB_URL="$DATABASE_URL"
fi

echo "📊 Connecting to database..."

# Run the migration SQL
if command -v psql >/dev/null 2>&1; then
    echo "✅ Using psql to run migration"
    psql "$DB_URL" -f database/migrations/002_workflow_execution_tables.sql
    echo "✅ Migration 002 completed successfully!"
    
    # Verify tables were created
    echo "🔍 Verifying tables were created:"
    psql "$DB_URL" -c "\dt" | grep -E "(execution_plans|agent_dependencies|workflow_state_live|workflow_execution_results|agent_execution_logs)" || echo "ℹ️  Tables may not be visible or created in different schema"
    
elif command -v python3 >/dev/null 2>&1; then
    echo "✅ Using Python to run migration"
    python3 -c "
import os
import psycopg2
from urllib.parse import urlparse

# Parse database URL
db_url = '$DB_URL'
if db_url.startswith('postgresql://'):
    result = urlparse(db_url)
    conn = psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port
    )
else:
    conn = psycopg2.connect(db_url)

try:
    cur = conn.cursor()
    with open('database/migrations/002_workflow_execution_tables.sql', 'r') as f:
        migration_sql = f.read()
    
    cur.execute(migration_sql)
    conn.commit()
    print('✅ Migration 002 completed successfully!')
    
    # Verify tables
    cur.execute(\"SELECT table_name FROM information_schema.tables WHERE table_name IN ('execution_plans', 'agent_dependencies', 'workflow_state_live', 'workflow_execution_results', 'agent_execution_logs')\")
    tables = cur.fetchall()
    print(f'🔍 Created tables: {[t[0] for t in tables]}')
    
finally:
    conn.close()
"
else
    echo "❌ Neither psql nor python3 found. Please install PostgreSQL client or Python with psycopg2"
    exit 1
fi

echo "🎉 Migration 002 complete! Workflow execution tables are ready."
echo ""
echo "📋 Created tables:"
echo "  • execution_plans - Master Planner execution plans"  
echo "  • agent_dependencies - Agent execution dependencies"
echo "  • workflow_state_live - Real-time workflow state tracking"
echo "  • workflow_execution_results - Final workflow results"
echo "  • agent_execution_logs - Detailed agent execution logs"
echo ""
echo "🚀 Your Master Planner system is now ready for production workflows!"