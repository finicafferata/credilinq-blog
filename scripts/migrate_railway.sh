#!/bin/bash
# Railway Database Migration Script
# Run this locally with your Railway DATABASE_URL

echo "🚀 Railway Database Migration Script"
echo "======================================"

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL environment variable is not set"
    echo ""
    echo "To fix this:"
    echo "1. Go to Railway Dashboard → PostgreSQL → Variables"
    echo "2. Copy the DATABASE_URL value"
    echo "3. Run: export DATABASE_URL='your-database-url-here'"
    echo "4. Run this script again"
    exit 1
fi

echo "✅ DATABASE_URL is set"
echo ""

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo "❌ ERROR: npx is not installed"
    echo "Please install Node.js and npm first"
    exit 1
fi

echo "📦 Generating Prisma Client..."
npx prisma generate

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate Prisma client"
    exit 1
fi

echo ""
echo "🔄 Pushing schema to database..."
npx prisma db push --skip-generate

if [ $? -ne 0 ]; then
    echo "❌ Failed to push schema to database"
    echo "Make sure your DATABASE_URL is correct and the database is accessible"
    exit 1
fi

echo ""
echo "✅ Database migration completed successfully!"
echo ""
echo "📊 Your database now has the following tables:"
echo "  - BlogPost (for blog content)"
echo "  - Campaign (for marketing campaigns)"
echo "  - CampaignTask (for campaign tasks)"
echo "  - Document (for knowledge base)"
echo "  - AgentPerformance (for agent metrics)"
echo "  - And more..."
echo ""
echo "🎉 Your Railway backend should now work with real data!"