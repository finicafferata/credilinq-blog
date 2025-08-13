-- Safe Competitor Intelligence Migration
-- This script adds only the new CI tables without modifying existing tables

-- Create CI enums
DO $$ BEGIN
    CREATE TYPE "CIContentType" AS ENUM ('blog_post', 'social_media_post', 'video', 'podcast', 'whitepaper', 'case_study', 'webinar', 'email_newsletter', 'press_release', 'product_update');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "CIPlatform" AS ENUM ('website', 'linkedin', 'twitter', 'facebook', 'instagram', 'youtube', 'medium', 'substack', 'tiktok', 'reddit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "CIIndustry" AS ENUM ('fintech', 'saas', 'ecommerce', 'healthcare', 'education', 'marketing', 'technology', 'finance', 'retail', 'media');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "CITrendStrength" AS ENUM ('weak', 'moderate', 'strong', 'viral');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "CIAlertPriority" AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "CICompetitorTier" AS ENUM ('direct', 'indirect', 'aspirational', 'adjacent');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create CI tables
CREATE TABLE IF NOT EXISTS "ci_competitors" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "name" TEXT NOT NULL,
    "domain" TEXT NOT NULL,
    "tier" "CICompetitorTier" NOT NULL,
    "industry" "CIIndustry" NOT NULL,
    "description" TEXT NOT NULL,
    "platforms" "CIPlatform"[],
    "monitoringKeywords" TEXT[],
    "lastMonitored" TIMESTAMPTZ(6),
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "metadata" JSONB,
    "createdAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ci_competitors_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_content_items" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "competitorId" UUID NOT NULL,
    "title" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "contentType" "CIContentType" NOT NULL,
    "platform" "CIPlatform" NOT NULL,
    "url" TEXT NOT NULL,
    "publishedAt" TIMESTAMPTZ(6) NOT NULL,
    "discoveredAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "author" TEXT,
    "engagementMetrics" JSONB,
    "keywords" TEXT[],
    "sentimentScore" DOUBLE PRECISION,
    "qualityScore" DOUBLE PRECISION,
    "metadata" JSONB,

    CONSTRAINT "ci_content_items_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_trends" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "topic" TEXT NOT NULL,
    "keywords" TEXT[],
    "industry" "CIIndustry" NOT NULL,
    "strength" "CITrendStrength" NOT NULL,
    "growthRate" DOUBLE PRECISION NOT NULL,
    "firstDetected" TIMESTAMPTZ(6) NOT NULL,
    "lastUpdated" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "peakDate" TIMESTAMPTZ(6),
    "competitorsUsing" TEXT[],
    "opportunityScore" DOUBLE PRECISION,
    "metadata" JSONB,

    CONSTRAINT "ci_trends_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_content_gaps" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "topic" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "opportunityScore" DOUBLE PRECISION NOT NULL,
    "difficultyScore" DOUBLE PRECISION NOT NULL,
    "potentialReach" INTEGER NOT NULL,
    "contentTypesMissing" "CIContentType"[],
    "platformsMissing" "CIPlatform"[],
    "keywords" TEXT[],
    "competitorsCovering" TEXT[],
    "suggestedApproach" TEXT,
    "identifiedAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB,

    CONSTRAINT "ci_content_gaps_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_competitor_insights" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "competitorId" UUID NOT NULL,
    "insightType" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "confidenceScore" DOUBLE PRECISION NOT NULL,
    "impactLevel" TEXT NOT NULL,
    "supportingEvidence" TEXT[],
    "recommendations" TEXT[],
    "generatedAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMPTZ(6),
    "metadata" JSONB,

    CONSTRAINT "ci_competitor_insights_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_market_analysis" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "industry" "CIIndustry" NOT NULL,
    "analysisStartDate" TIMESTAMPTZ(6) NOT NULL,
    "analysisEndDate" TIMESTAMPTZ(6) NOT NULL,
    "totalContentAnalyzed" INTEGER NOT NULL,
    "topTopics" JSONB NOT NULL,
    "contentTypeDistribution" JSONB NOT NULL,
    "platformDistribution" JSONB NOT NULL,
    "engagementBenchmarks" JSONB NOT NULL,
    "trendingKeywords" TEXT[],
    "contentVelocity" DOUBLE PRECISION NOT NULL,
    "qualityTrends" JSONB NOT NULL,
    "generatedAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB,

    CONSTRAINT "ci_market_analysis_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_alerts" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "alertType" TEXT NOT NULL,
    "priority" "CIAlertPriority" NOT NULL,
    "title" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "data" JSONB NOT NULL,
    "competitorIds" TEXT[],
    "trendIds" TEXT[],
    "contentIds" TEXT[],
    "createdAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sentAt" TIMESTAMPTZ(6),
    "acknowledgedAt" TIMESTAMPTZ(6),
    "expiresAt" TIMESTAMPTZ(6),
    "recipients" TEXT[],
    "metadata" JSONB,

    CONSTRAINT "ci_alerts_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "ci_alert_subscriptions" (
    "id" UUID NOT NULL DEFAULT extensions.uuid_generate_v4(),
    "userId" TEXT NOT NULL,
    "alertTypes" TEXT[],
    "competitors" TEXT[],
    "keywords" TEXT[],
    "priorityThreshold" "CIAlertPriority" NOT NULL DEFAULT 'medium',
    "deliveryChannels" TEXT[],
    "frequencyLimit" INTEGER NOT NULL DEFAULT 10,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB,

    CONSTRAINT "ci_alert_subscriptions_pkey" PRIMARY KEY ("id")
);

-- Create junction tables for many-to-many relationships
CREATE TABLE IF NOT EXISTS "_CITrendContent" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CIGapContent" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CITrendGaps" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CITrendAlerts" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CIGapAlerts" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CIInsightAlerts" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

CREATE TABLE IF NOT EXISTS "_CIAlertSubscriptions" (
    "A" UUID NOT NULL,
    "B" UUID NOT NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS "ci_competitors_industry_idx" ON "ci_competitors"("industry");
CREATE INDEX IF NOT EXISTS "ci_competitors_tier_idx" ON "ci_competitors"("tier");
CREATE INDEX IF NOT EXISTS "ci_competitors_isActive_idx" ON "ci_competitors"("isActive");

CREATE INDEX IF NOT EXISTS "ci_content_items_competitorId_idx" ON "ci_content_items"("competitorId");
CREATE INDEX IF NOT EXISTS "ci_content_items_contentType_idx" ON "ci_content_items"("contentType");
CREATE INDEX IF NOT EXISTS "ci_content_items_platform_idx" ON "ci_content_items"("platform");
CREATE INDEX IF NOT EXISTS "ci_content_items_publishedAt_idx" ON "ci_content_items"("publishedAt");
CREATE INDEX IF NOT EXISTS "ci_content_items_discoveredAt_idx" ON "ci_content_items"("discoveredAt");

CREATE INDEX IF NOT EXISTS "ci_trends_industry_idx" ON "ci_trends"("industry");
CREATE INDEX IF NOT EXISTS "ci_trends_strength_idx" ON "ci_trends"("strength");
CREATE INDEX IF NOT EXISTS "ci_trends_firstDetected_idx" ON "ci_trends"("firstDetected");
CREATE INDEX IF NOT EXISTS "ci_trends_lastUpdated_idx" ON "ci_trends"("lastUpdated");

CREATE INDEX IF NOT EXISTS "ci_content_gaps_opportunityScore_idx" ON "ci_content_gaps"("opportunityScore");
CREATE INDEX IF NOT EXISTS "ci_content_gaps_difficultyScore_idx" ON "ci_content_gaps"("difficultyScore");
CREATE INDEX IF NOT EXISTS "ci_content_gaps_identifiedAt_idx" ON "ci_content_gaps"("identifiedAt");

CREATE INDEX IF NOT EXISTS "ci_competitor_insights_competitorId_idx" ON "ci_competitor_insights"("competitorId");
CREATE INDEX IF NOT EXISTS "ci_competitor_insights_insightType_idx" ON "ci_competitor_insights"("insightType");
CREATE INDEX IF NOT EXISTS "ci_competitor_insights_confidenceScore_idx" ON "ci_competitor_insights"("confidenceScore");
CREATE INDEX IF NOT EXISTS "ci_competitor_insights_generatedAt_idx" ON "ci_competitor_insights"("generatedAt");

CREATE INDEX IF NOT EXISTS "ci_market_analysis_industry_idx" ON "ci_market_analysis"("industry");
CREATE INDEX IF NOT EXISTS "ci_market_analysis_generatedAt_idx" ON "ci_market_analysis"("generatedAt");
CREATE INDEX IF NOT EXISTS "ci_market_analysis_analysisStartDate_idx" ON "ci_market_analysis"("analysisStartDate");

CREATE INDEX IF NOT EXISTS "ci_alerts_alertType_idx" ON "ci_alerts"("alertType");
CREATE INDEX IF NOT EXISTS "ci_alerts_priority_idx" ON "ci_alerts"("priority");
CREATE INDEX IF NOT EXISTS "ci_alerts_createdAt_idx" ON "ci_alerts"("createdAt");
CREATE INDEX IF NOT EXISTS "ci_alerts_sentAt_idx" ON "ci_alerts"("sentAt");

CREATE INDEX IF NOT EXISTS "ci_alert_subscriptions_userId_idx" ON "ci_alert_subscriptions"("userId");
CREATE INDEX IF NOT EXISTS "ci_alert_subscriptions_isActive_idx" ON "ci_alert_subscriptions"("isActive");
CREATE INDEX IF NOT EXISTS "ci_alert_subscriptions_priorityThreshold_idx" ON "ci_alert_subscriptions"("priorityThreshold");

-- Add foreign key constraints
DO $$ BEGIN
    ALTER TABLE "ci_content_items" ADD CONSTRAINT "ci_content_items_competitorId_fkey" FOREIGN KEY ("competitorId") REFERENCES "ci_competitors"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "ci_competitor_insights" ADD CONSTRAINT "ci_competitor_insights_competitorId_fkey" FOREIGN KEY ("competitorId") REFERENCES "ci_competitors"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Add unique constraints for junction tables
DO $$ BEGIN
    ALTER TABLE "_CITrendContent" ADD CONSTRAINT "_CITrendContent_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CIGapContent" ADD CONSTRAINT "_CIGapContent_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CITrendGaps" ADD CONSTRAINT "_CITrendGaps_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CITrendAlerts" ADD CONSTRAINT "_CITrendAlerts_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CIGapAlerts" ADD CONSTRAINT "_CIGapAlerts_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CIInsightAlerts" ADD CONSTRAINT "_CIInsightAlerts_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "_CIAlertSubscriptions" ADD CONSTRAINT "_CIAlertSubscriptions_AB_unique" UNIQUE ("A", "B");
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;