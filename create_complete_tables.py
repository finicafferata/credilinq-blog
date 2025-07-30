#!/usr/bin/env python3
"""
Script para crear todas las tablas según el esquema de Prisma.
"""

import psycopg2
import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_complete_tables():
    """Crear todas las tablas según el esquema de Prisma."""
    load_dotenv()
    
    try:
        # Conectar a la base de datos
        database_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        logger.info("🔗 Conectado a Supabase")
        
        # Crear enums
        logger.info("📝 Creando enums...")
        cursor.execute("""
            DO $$ BEGIN
                CREATE TYPE "PostStatus" AS ENUM (
                    'draft', 'published', 'archived', 'deleted', 'edited'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        cursor.execute("""
            DO $$ BEGIN
                CREATE TYPE "TaskStatus" AS ENUM (
                    'pending', 'in_progress', 'needs_review', 'approved', 'rejected', 'completed', 'error'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        # Crear tabla BlogPost (reemplazar blogs)
        logger.info("📝 Creando tabla 'BlogPost'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "BlogPost" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title VARCHAR(255) NOT NULL,
                "contentMarkdown" TEXT,
                "initialPrompt" JSONB,
                status "PostStatus" DEFAULT 'draft',
                "geoMetadata" JSONB,
                "geoOptimized" BOOLEAN DEFAULT false,
                "geoScore" INTEGER,
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Crear tabla Campaign
        logger.info("📝 Creando tabla 'Campaign'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Campaign" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "blogPostId" UUID UNIQUE REFERENCES "BlogPost"(id)
            );
        """)
        
        # Crear tabla Briefing
        logger.info("📝 Creando tabla 'Briefing'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Briefing" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "campaignName" VARCHAR(255) NOT NULL,
                "marketingObjective" TEXT NOT NULL,
                "targetAudience" TEXT NOT NULL,
                channels JSONB NOT NULL,
                "desiredTone" VARCHAR(255) NOT NULL,
                language VARCHAR(50) NOT NULL,
                "companyContext" TEXT,
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "campaignId" UUID UNIQUE REFERENCES "Campaign"(id)
            );
        """)
        
        # Crear tabla ContentStrategy
        logger.info("📝 Creando tabla 'ContentStrategy'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "ContentStrategy" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "campaignName" VARCHAR(255) NOT NULL,
                "narrativeApproach" TEXT NOT NULL,
                hooks JSONB NOT NULL,
                themes JSONB NOT NULL,
                "toneByChannel" JSONB NOT NULL,
                "keyPhrases" JSONB NOT NULL,
                notes TEXT,
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "campaignId" UUID UNIQUE REFERENCES "Campaign"(id)
            );
        """)
        
        # Crear tabla CampaignTask (actualizar la existente)
        logger.info("📝 Actualizando tabla 'CampaignTask'...")
        cursor.execute("""
            DROP TABLE IF EXISTS "campaign_tasks" CASCADE;
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "CampaignTask" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "campaignId" UUID NOT NULL REFERENCES "Campaign"(id) ON DELETE CASCADE,
                "taskType" VARCHAR(100) NOT NULL,
                "targetFormat" VARCHAR(100),
                "targetAsset" VARCHAR(255),
                status VARCHAR(50) NOT NULL,
                result TEXT,
                "imageUrl" TEXT,
                error TEXT,
                "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Crear tabla Document (actualizar la existente)
        logger.info("📝 Actualizando tabla 'Document'...")
        cursor.execute("""
            DROP TABLE IF EXISTS "documents" CASCADE;
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Document" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title VARCHAR(255) NOT NULL,
                "storagePath" TEXT NOT NULL,
                "uploadedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Crear tabla DocumentChunk
        logger.info("📝 Creando tabla 'DocumentChunk'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "DocumentChunk" (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT,
                "documentId" UUID REFERENCES "Document"(id) ON DELETE CASCADE
            );
        """)
        
        # Crear índices para mejor rendimiento
        logger.info("📝 Creando índices...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blogpost_status ON "BlogPost"(status);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blogpost_created_at ON "BlogPost"("createdAt");')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaign_blogpost_id ON "Campaign"("blogPostId");')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaigntask_campaign_id ON "CampaignTask"("campaignId");')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documentchunk_document_id ON "DocumentChunk"("documentId");')
        
        # Commit los cambios
        conn.commit()
        
        logger.info("✅ Todas las tablas creadas exitosamente")
        
        # Verificar las tablas creadas
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        logger.info(f"📋 Tablas disponibles ({len(tables)}):")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al crear las tablas: {e}")
        return False

def main():
    """Función principal."""
    logger.info("🚀 Creando tablas completas según esquema de Prisma...")
    logger.info("=" * 60)
    
    success = create_complete_tables()
    
    if success:
        logger.info("🎉 ¡Base de datos configurada correctamente!")
        logger.info("💡 Ahora puedes ejecutar tu aplicación")
    else:
        logger.error("❌ Error al configurar la base de datos")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 