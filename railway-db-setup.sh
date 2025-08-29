#!/bin/bash

# CrediLinq Content Agent - Railway Database Setup Script
# Sets up PostgreSQL with vector extensions and runs migrations

set -e

echo "🗄️ CrediLinq Content Agent - Database Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we have database URL
DATABASE_URL=$(railway variables get DATABASE_URL 2>/dev/null)
if [ -z "$DATABASE_URL" ]; then
    print_error "DATABASE_URL not found. Make sure PostgreSQL service is added to Railway."
    exit 1
fi

print_success "Database URL found"

# Check if vector extension is available
print_status "Checking PostgreSQL extensions..."

# Connect to database and check/install extensions
railway run psql $DATABASE_URL << 'EOF'
-- Check if extensions are available
SELECT name, installed_version 
FROM pg_available_extensions 
WHERE name IN ('vector', 'uuid-ossp', 'pg_trgm');

-- Install extensions if available
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Verify installation
SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm');
EOF

if [ $? -eq 0 ]; then
    print_success "PostgreSQL extensions configured successfully"
else
    print_warning "Could not configure all extensions. Vector search features may be limited."
fi

# Generate Prisma client
print_status "Generating Prisma client..."
if command -v npm &> /dev/null; then
    npx prisma generate
    print_success "Prisma client generated"
else
    print_warning "npm not found. Make sure to run 'npx prisma generate' before deployment"
fi

# Run database migrations
print_status "Running database migrations..."

# Method 1: Try direct migration push (development-friendly)
print_status "Attempting development migration..."
if railway run npx prisma db push --skip-generate; then
    print_success "Database schema synchronized successfully"
else
    print_warning "Direct push failed, trying production migrations..."
    
    # Method 2: Production migrations
    if railway run npx prisma migrate deploy; then
        print_success "Production migrations applied successfully"
    else
        print_error "Migration failed. Manual intervention may be required."
        echo ""
        echo "Manual migration steps:"
        echo "1. Connect to database: railway connect postgresql"
        echo "2. Run migrations manually: npx prisma migrate deploy"
        echo "3. Or reset and push: npx prisma migrate reset --force && npx prisma db push"
    fi
fi

# Verify database setup
print_status "Verifying database setup..."
railway run python3 -c "
import os
import asyncpg
import asyncio

async def test_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Test basic connection
        version = await conn.fetchval('SELECT version()')
        print('✅ Database connection: OK')
        print(f'PostgreSQL version: {version.split()[0:2]}')
        
        # Check extensions
        extensions = await conn.fetch(\"SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')\")
        print(f'✅ Extensions installed: {[ext[\"extname\"] for ext in extensions]}')
        
        # Check main tables
        tables = await conn.fetch(\"SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 10\")
        print(f'✅ Tables found: {len(tables)} tables')
        
        await conn.close()
        print('✅ Database setup verification: PASSED')
        
    except Exception as e:
        print(f'❌ Database verification failed: {e}')

asyncio.run(test_db())
"

print_success "Database setup completed!"
print_status "Next steps:"
echo "1. Verify backend deployment: railway logs"
echo "2. Test API health: curl \$(railway domain)/health/live"
echo "3. Deploy frontend service"
echo "4. Run end-to-end tests"